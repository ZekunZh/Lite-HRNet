dataset_info = dict(
    dataset_name='gleamer_foot_frontal',
    paper_info=dict(
        author='ZHANG Zekun',
        title='Keypoint dataset for Gleamer',
        year="2022",
        homepage="https://www.gleamer.ai/"
    ),
    keypoint_info={
        0: dict(name='M_1_PROXIMAL_LEFT', id=0, color=[51, 153, 255], swap='M_1_PROXIMAL_RIGHT'),
        1: dict(name='M_1_PROXIMAL_RIGHT', id=0, color=[51, 153, 255], swap='M_1_PROXIMAL_LEFT'),
        2: dict(name='M_1_DISTAL_LEFT', id=0, color=[51, 153, 255], swap='M_1_DISTAL_RIGHT'),
        3: dict(name='M_1_DISTAL_RIGHT', id=0, color=[0, 255, 0], swap='M_1_DISTAL_LEFT'),
        4: dict(name='P_1_PROXIMAL_LEFT', id=0, color=[0, 255, 0], swap='P_1_PROXIMAL_RIGHT'),
        5: dict(name='P_1_PROXIMAL_RIGHT', id=0, color=[0, 255, 0], swap='P_1_PROXIMAL_LEFT'),
        6: dict(name='P_1_DISTAL_LEFT', id=0, color=[0, 255, 0], swap='P_1_DISTAL_RIGHT'),
        7: dict(name='P_1_DISTAL_RIGHT', id=0, color=[255, 128, 0], swap='P_1_DISTAL_LEFT'),
        8: dict(name='M_2_PROXIMAL_LEFT', id=0, color=[255, 128, 0], swap='M_2_PROXIMAL_RIGHT'),
        9: dict(name='M_2_PROXIMAL_RIGHT', id=0, color=[255, 128, 0], swap='M_2_PROXIMAL_LEFT'),
        10: dict(name='M_2_DISTAL_LEFT', id=0, color=[255, 128, 0], swap='M_2_DISTAL_RIGHT'),
        11: dict(name='M_2_DISTAL_RIGHT', id=0, color=[255, 128, 0], swap='M_2_DISTAL_LEFT'),
        12: dict(name='M_5_PROXIMAL_LEFT', id=0, color=[255, 128, 0], swap='M_5_PROXIMAL_RIGHT'),
        13: dict(name='M_5_PROXIMAL_RIGHT', id=0, color=[255, 128, 0], swap='M_5_PROXIMAL_LEFT'),
        14: dict(name='M_5_DISTAL_LEFT', id=0, color=[255, 128, 0], swap='M_5_DISTAL_RIGHT'),
        15: dict(name='M_5_DISTAL_RIGHT', id=0, color=[255, 128, 0], swap='M_5_DISTAL_LEFT'),
    },
    skeleton_info={
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ],
    sigmas=[
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    ])
